import tensorflow.contrib.eager as tfe
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tfe.enable_eager_execution(device_policy=tfe.DEVICE_PLACEMENT_SILENT)


class RegressionRNN(tf.keras.Model):
    def __init__(self, cell_size=64, dense_size=128, predict_ahead=1,
                 device='cpu:0', checkpoint_directory=None):
        super(RegressionRNN, self).__init__()
        w_initializer = tf.contrib.layers.xavier_initializer()
        b_initializer = tf.zeros_initializer()
        self.dense_layer = tf.keras.layers.Dense(dense_size, activation=tf.nn.relu,
                                                 kernel_initializer=w_initializer,
                                                 bias_initializer=b_initializer)
        self.pred_layer = tf.keras.layers.Dense(predict_ahead * 24, activation=None,
                                                kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer)
        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)

        self.device = device
        self.checkpoint_directory = checkpoint_directory

    def predict(self, X):
        num_samples = tf.shape(X)[0]
        #初始h
        state = self.rnn_cell.zero_state(num_samples, dtype=tf.float32)
        unstacked_input = tf.unstack(X, axis=1)
        for input_step in unstacked_input:
            output, state = self.rnn_cell(input_step, state)
        dense = self.dense_layer(output)
        preds = self.pred_layer(dense)
        return preds

    def loss_fn(self, X, y):
        preds = self.predict(X)
        loss = tf.losses.mean_squared_error(y, preds)
        return loss

    def grads_fn(self, X, y):
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(X, y)
        return tape.gradient(loss, self.variables)

    def restore_model(self):
        with tf.device(self.device):
            # Run the model once to initialize variables
            dummy_input = tf.constant(tf.zeros((1, 5, 24)))
            dummy_pred = self.predict(dummy_input)
            # Restore the variables of the model
            saver = tfe.Saver(self.variables)
            saver.restore(tf.train.latest_checkpoint
                          (self.checkpoint_directory))

    def save_model(self, global_step=0):
        tfe.Saver(self.variables).save(self.checkpoint_directory,
                                       global_step=global_step)

    def fit(self, training_data, eval_data, optimizer, num_epochs=500,
            early_stopping_rounds=10, verbose=10, train_from_scratch=False):

        if train_from_scratch == False:
            self.restore_model()
        best_loss = 999
        train_loss = tfe.metrics.Mean('train_loss')
        eval_loss = tfe.metrics.Mean('eval_loss')

        self.history = {}
        self.history['train_loss'] = []
        self.history['eval_loss'] = []

        with tf.device(self.device):
            for i in range(num_epochs):
                for X, y in tfe.Iterator(training_data):
                    grads = self.grads_fn(X, y)
                    optimizer.apply_gradients(zip(grads, self.variables))
                for X, y in tfe.Iterator(training_data):
                    loss = self.loss_fn(X, y)
                    train_loss(loss)
                self.history['train_loss'].append(train_loss.result().numpy())
                train_loss.init_variables()

                for X, y in tfe.Iterator(eval_data):
                    loss = self.loss_fn(X, y)
                    eval_loss(loss)
                self.history['eval_loss'].append(eval_loss.result().numpy())
                eval_loss.init_variables()


                if (i == 0) | ((i + 1) % verbose == 0):
                    print('Train loss at epoch %d: ' % (i + 1), self.history['train_loss'][-1])
                    print('Eval loss at epoch %d: ' % (i + 1), self.history['eval_loss'][-1])

                if self.history['eval_loss'][-1] < best_loss:
                    best_loss = self.history['eval_loss'][-1]
                    count = early_stopping_rounds
                else:
                    count -= 1
                if count == 0:
                    break


def data_processing():
    energy_df = pd.read_csv('dataset/load_forecasting/spain_hourly_entsoe.csv')
    energy_df.columns = ['time', 'forecasted_load', 'actual_load']
    energy_df['date'] = energy_df['time'].apply(lambda x: dt.strptime(x.split('-')[0].strip(), '%d.%m.%Y %H:%M'))

    # Split dataset into train and test
    train_size = 0.8
    end_train = int(len(energy_df) * train_size / 24) * 24
    train_energy_df = energy_df.iloc[:end_train, :]
    test_energy_df = energy_df.iloc[end_train:, :]
    # plt.figure(figsize=(14, 6))
    # plt.plot(train_energy_df['date'], train_energy_df['actual_load'], color='cornflowerblue')
    # plt.title('Train dataset', fontsize=17)
    # plt.ylabel('Energy Demand [MW]')
    # plt.show()
    # Interpolate missing measurements
    train_energy_df = train_energy_df.interpolate(limit_direction='both')
    test_energy_df = test_energy_df.interpolate(limit_direction='both')
    scaler = StandardScaler().fit(train_energy_df['actual_load'][:, None])
    train_energy_df['actual_load'] = scaler.transform(train_energy_df['actual_load'][:, None])
    test_energy_df['actual_load'] = scaler.transform(test_energy_df['actual_load'][:, None])
    return train_energy_df,test_energy_df

def moving_window_samples(timeseries, look_back=5, predict_ahead=1):
    '''
    Function to create input and target samples from a time-series
    using a lag of one day.
    使用前五天的数据预测第六天的结果
    Args:
        timeseries: timeseries dataset.
        look_back: 输入的大小. 特指利用前几天的数据.
        predict_ahead: 输出的大小. 指预测几天的数据.

    Returns:样本对（前五天的数据，第六天的数据）
    '''

    n_strides = int((len(timeseries) - predict_ahead * 24 - look_back * 24 + 24) / 24)
    input_samples = np.zeros((n_strides, look_back * 24))
    target_samples = np.zeros((n_strides, predict_ahead * 24))
    for i in range(n_strides):
        end_input = i * 24 + look_back * 24
        input_samples[i, :] = timeseries[i * 24:end_input]
        target_samples[i, :] = timeseries[end_input:(end_input + predict_ahead * 24)]
    # Reshape input to (num_samples, timesteps, input_dimension)
    input_samples = input_samples.reshape((-1, look_back, 24))
    return input_samples.astype('float32'), target_samples.astype('float32')

if __name__ == '__main__':
    train_energy_df, test_energy_df = data_processing()
    train_input_samples, train_target_samples = moving_window_samples(train_energy_df['actual_load'],look_back=5, predict_ahead=1)
    test_input_samples, test_target_samples = moving_window_samples(test_energy_df['actual_load'],look_back=5, predict_ahead=1)

    batch_size = 64
    train_dataset = (tf.data.Dataset.from_tensor_slices((train_input_samples, train_target_samples)).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices((test_input_samples, test_target_samples)).batch(batch_size))
    checkpoint_directory = 'models_checkpoints/DemandRNN/'
    device = 'gpu:0' if tfe.num_gpus() > 0 else 'cpu:0'
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    model = RegressionRNN(cell_size=16, dense_size=16, predict_ahead=1,device=device, checkpoint_directory=checkpoint_directory)
    model.fit(train_dataset, test_dataset, optimizer, num_epochs=500,
              early_stopping_rounds=10, verbose=50, train_from_scratch=True)
    model.save_model()
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.history['train_loss'])), model.history['train_loss'],
             color='cornflowerblue', label='Train loss')
    plt.plot(range(len(model.history['eval_loss'])), model.history['eval_loss'],
             color='gold', label='Test loss')
    plt.title('Model performance during training', fontsize=15)
    plt.xlabel('Number of epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    '''
    Train loss at epoch 1:  0.4270984549075365
    Eval loss at epoch 1:  0.4990002289414406
    Train loss at epoch 50:  0.025285931886173785
    Eval loss at epoch 50:  0.044382193591445684
    '''