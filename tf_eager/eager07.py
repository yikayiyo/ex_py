#-*- encoding=utf8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
class ERCNN(tf.keras.Model):
    def __init__(self,num_classes, device='cpu:0', checkpoint_directory=None):
        super().__init__()
        # Initialize layers
        self.conv1 = tf.layers.Conv2D(16, 5, padding='same', activation=None)
        self.batch1 = tf.layers.BatchNormalization()
        self.conv2 = tf.layers.Conv2D(16, 5, 2, padding='same', activation=None)
        self.batch2 = tf.layers.BatchNormalization()
        self.conv3 = tf.layers.Conv2D(32, 5, padding='same', activation=None)
        self.batch3 = tf.layers.BatchNormalization()
        self.conv4 = tf.layers.Conv2D(32, 5, 2, padding='same', activation=None)
        self.batch4 = tf.layers.BatchNormalization()
        self.conv5 = tf.layers.Conv2D(64, 3, padding='same', activation=None)
        self.batch5 = tf.layers.BatchNormalization()
        self.conv6 = tf.layers.Conv2D(64, 3, 2, padding='same', activation=None)
        self.batch6 = tf.layers.BatchNormalization()
        self.conv7 = tf.layers.Conv2D(64, 1, padding='same', activation=None)
        self.batch7 = tf.layers.BatchNormalization()
        self.conv8 = tf.layers.Conv2D(128, 3, 2, padding='same', activation=None)
        self.batch8 = tf.keras.layers.BatchNormalization()
        self.conv9 = tf.layers.Conv2D(256, 1, padding='same', activation=None)
        self.batch9 = tf.keras.layers.BatchNormalization()
        self.conv10 = tf.layers.Conv2D(128, 3, 2, padding='same', activation=None)
        self.conv11 = tf.layers.Conv2D(256, 1, padding='same', activation=None)
        self.batch11 = tf.layers.BatchNormalization()
        self.conv12 = tf.layers.Conv2D(num_classes, 3, 2, padding='same', activation=None)

        self.device = device
        self.checkpoint_directory = checkpoint_directory

    def predict(self, images, training):
        """ Predicts the probability of each class, based on the input sample.

            Args:
                images: 4D tensor. Either an image or a batch of images.
                training: Boolean. Either the network is predicting in
                          training mode or not.
        """
        x = self.conv1(images)
        x = self.batch1(x, training=training)
        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.4, training=training)
        x = self.conv3(x)
        x = self.batch3(x, training=training)
        x = self.conv4(x)
        x = self.batch4(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.3, training=training)
        x = self.conv5(x)
        x = self.batch5(x, training=training)
        x = self.conv6(x)
        x = self.batch6(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.3, training=training)
        x = self.conv7(x)
        x = self.batch7(x, training=training)
        x = self.conv8(x)
        x = self.batch8(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=0.3, training=training)
        x = self.conv9(x)
        x = self.batch9(x, training=training)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.batch11(x, training=training)
        x = self.conv12(x)
        return tf.layers.flatten(x)

    def loss_fn(self, images, target, training):
        """ Defines the loss function used during
            training.
        """
        preds = self.predict(images, training)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=preds)
        return loss

    def grads_fn(self,images,target,training):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(images, target, training)
        return tape.gradient(loss, self.variables)

    def restore_model(self):
        with tf.device(self.device):
            # Run the model once to initialize variables
            dummy_input = tf.constant(tf.zeros((1, 48, 48, 1)))
            dummy_pred = self.predict(dummy_input, training=False)
            # Restore the variables of the model
            saver = tfe.Saver(self.variables)
            saver.restore(tf.train.latest_checkpoint
                          (self.checkpoint_directory))

    def save_model(self, global_step=0):
        tfe.Saver(self.variables).save(self.checkpoint_directory,
                                       global_step=global_step)

    def compute_accuracy(self, input_data):
        with tf.device(self.device):
            acc = tfe.metrics.Accuracy()
            for images, targets in tfe.Iterator(input_data):
                # Predict the probability of each class
                logits = self.predict(images, training=False)
                # Select the class with the highest probability
                preds = tf.argmax(logits, axis=1)
                # Compute the accuracy
                acc(tf.reshape(targets, [-1, ]), preds)
        return acc

    def fit(self, training_data, eval_data, optimizer, num_epochs=500,
            early_stopping_rounds=10, verbose=10, train_from_scratch=False):
        if train_from_scratch == False:
            self.restore_model()
        # Initialize best loss. This variable will store the lowest loss on the
        # eval dataset.
        best_loss = 999
        train_loss = tfe.metrics.Mean('train_loss')
        eval_loss = tfe.metrics.Mean('eval_loss')
        self.history = {}
        self.history['train_loss'] = []
        self.history['eval_loss'] = []
        # Begin training
        with tf.device(self.device):
            for i in range(num_epochs):
                # Training with gradient descent
                for images, target in tfe.Iterator(training_data):
                    grads = self.grads_fn(images, target, True)
                    optimizer.apply_gradients(zip(grads, self.variables))

                # Compute the loss on the training data after one epoch
                for images, target in tfe.Iterator(training_data):
                    loss = self.loss_fn(images, target, False)
                    train_loss(loss)
                self.history['train_loss'].append(train_loss.result().numpy())
                # Reset metrics
                train_loss.init_variables()

                # Compute the loss on the eval data after one epoch
                for images, target in tfe.Iterator(eval_data):
                    loss = self.loss_fn(images, target, False)
                    eval_loss(loss)
                self.history['eval_loss'].append(eval_loss.result().numpy())
                # Reset metrics
                eval_loss.init_variables()

                # Print train and eval losses
                if (i == 0) | ((i + 1) % verbose == 0):
                    print('Train loss at epoch %d: ' % (i + 1), self.history['train_loss'][-1])
                    print('Eval loss at epoch %d: ' % (i + 1), self.history['eval_loss'][-1])

                # Check for early stopping
                if self.history['eval_loss'][-1] < best_loss:
                    best_loss = self.history['eval_loss'][-1]
                    count = early_stopping_rounds
                else:
                    count -= 1
                if count == 0:
                    break


def data_process():
    path_data = 'dataset/fer2013/fer2013.csv'
    data = pd.read_csv(path_data)
    data['pixels']=data['pixels'].apply(lambda x:[int(i)for i in x.split()])

    data_train = data[data['Usage']=='Training']
    size_train = data_train.shape[0]
    print('Number samples in the training dataset: ', size_train)

    data_dev = data[data['Usage']!='Training']
    size_dev = data_dev.shape[0]
    print('Number samples in the development dataset: ', size_dev)

    X_train, y_train = data_train['pixels'].tolist(), data_train['emotion'].values
    X_train = np.array(X_train, dtype='float32').reshape(-1,48,48,1)
    X_train = X_train/255.0

    X_dev, y_dev = data_dev['pixels'].tolist(), data_dev['emotion'].values
    X_dev = np.array(X_dev, dtype='float32').reshape(-1,48,48,1)
    X_dev = X_dev/255.0

    batch_size = 64
    training_data = tf.data.Dataset.from_tensor_slices((X_train, y_train[:,None])).batch(batch_size)
    eval_data = tf.data.Dataset.from_tensor_slices((X_dev, y_dev[:,None])).batch(batch_size)
    return training_data,eval_data

def train():
    # Specify the path where you want to save/restore the trained variables.
    checkpoint_directory = 'models_checkpoints/EmotionCNN/'
    # Use the GPU if available.
    device = 'gpu:0' if tfe.num_gpus() > 0 else 'cpu:0'
    # Define optimizer.
    optimizer = tf.train.AdamOptimizer()

    # Instantiate model. This doesn't initialize the variables yet.
    model = ERCNN(num_classes=7, device=device,
                                  checkpoint_directory=checkpoint_directory)
    training_data,eval_data = data_process()
    # Train model
    model.fit(training_data, eval_data, optimizer, num_epochs=500,
              early_stopping_rounds=5, verbose=10, train_from_scratch=False)
    model.save_model()
    plt.plot(range(len(model.history['train_loss'])), model.history['train_loss'],
             color='b', label='Train loss')
    plt.plot(range(len(model.history['eval_loss'])), model.history['eval_loss'],
             color='r', label='Dev loss')
    plt.title('Model performance during training', fontsize=15)
    plt.xlabel('Number of epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    train_acc = model.compute_accuracy(training_data)
    eval_acc = model.compute_accuracy(eval_data)

    print('Train accuracy: ', train_acc.result().numpy())
    print('Eval accuracy: ', eval_acc.result().numpy())

if __name__ == '__main__':
    train()
