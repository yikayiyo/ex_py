import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
# Import functions to write and parse TFRecords
from data_utils import imdb2tfrecords
from data_utils import parse_imdb_sequence
import pickle
import nltk
from nltk.tokenize import word_tokenize
import re
tfe.enable_eager_execution()

'''
- 下载数据转换格式为TFRecord.
- 利用迭代器从磁盘中批量读取数据，并自动填充
- 构建一个单词层级的RNN模型,分别使用LSTM and UGRNN cells.
- 比较测试集上的结果.
- 保存恢复模型.
- 实验新评论的检测效果
'''

class RNNModel(tf.keras.Model):

    def __init__(self,embedding_size=100, cell_size=64, dense_size=128,
                 num_classes=2, vocabulary_size=None, rnn_cell='lstm',
                 device='cpu:0', checkpoint_directory=None):
        super(RNNModel, self).__init__()
        #指明参数初始化的方法
        w_initializer = tf.contrib.layers.xavier_initializer()
        b_initializer = tf.zeros_initializer()
        # 初始化word embeddings权重
        self.embeddings = tf.keras.layers.Embedding(vocabulary_size, embedding_size,
                                                    embeddings_initializer=w_initializer)
        # 中间层
        self.dense_layer = tf.keras.layers.Dense(dense_size, activation=tf.nn.relu,
                                                 kernel_initializer=w_initializer,
                                                 bias_initializer=b_initializer)

        # 输出层
        self.pred_layer = tf.keras.layers.Dense(num_classes, activation=None,
                                                kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer)

        # LSTM cell
        if rnn_cell == 'lstm':
            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)
        # Else UGRNN cell
        else:
            self.rnn_cell = tf.contrib.rnn.UGRNNCell(cell_size)

        self.device = device
        self.checkpoint_directory = checkpoint_directory

    def predict(self,X,seq_length,is_training):
        # 一个batch里的样本数
        num_samples = tf.shape(X)[0]
        # 初始化LSTM cell状态
        state = self.rnn_cell.zero_state(num_samples, dtype=tf.float32)
        # 获取序列中每个词的向量
        embedded_words = self.embeddings(X)
        # Unstack the embeddings-->(time_steps, cell_size)
        unstacked_embeddings = tf.unstack(embedded_words, axis=1)
        # Iterate through each timestep and append the predictions
        outputs = []
        for input_step in unstacked_embeddings:
            output, state = self.rnn_cell(input_step, state)
            outputs.append(output)
        # Stack outputs to (batch_size, time_steps, cell_size)
        outputs = tf.stack(outputs, axis=1)

        # 提取最后一个timestep的输出---idxs_last_output.shape-->(batchsize,2)
        idxs_last_output = tf.stack([tf.range(num_samples),
                                     tf.cast(seq_length - 1, tf.int32)], axis=1)
        final_output = tf.gather_nd(outputs, idxs_last_output)

        # Add dropout for regularization
        dropped_output = tf.layers.dropout(final_output, rate=0.3, training=is_training)

        # Pass the last cell state through a dense layer (ReLU activation)
        dense = self.dense_layer(dropped_output)
        # Compute the unnormalized log probabilities
        logits = self.pred_layer(dense)
        return logits


    def loss_fn(self, X, y, seq_length, is_training):
        preds = self.predict(X, seq_length, is_training)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=preds)
        return loss


    def grads_fn(self, X, y, seq_length, is_training):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(X, y, seq_length, is_training)
        return tape.gradient(loss, self.variables)


    def restore_model(self):

        with tf.device(self.device):
            dummy_input = tf.constant(tf.zeros((1, 1)))
            dummy_length = tf.constant(1, shape=(1,))
            dummy_pred = self.predict(dummy_input, dummy_length, False)
            saver = tfe.Saver(self.variables)
            saver.restore(tf.train.latest_checkpoint
                          (self.checkpoint_directory))

    def save_model(self,global_step=0):
        '''Function to save trained model'''
        tfe.Saver(self.variables).save(self.checkpoint_directory,
                                       global_step=global_step)

    def fit(self, training_data, eval_data, optimizer, num_epochs=500,
            early_stopping_rounds=10, verbose=10, train_from_scratch=False):

        if train_from_scratch == False:
            self.restore_model()
        best_acc = 0
        train_acc = tfe.metrics.Accuracy('train_acc')
        eval_acc = tfe.metrics.Accuracy('eval_acc')

        self.history = {}
        self.history['train_acc'] = []
        self.history['eval_acc'] = []

        with tf.device(self.device):
            for i in range(num_epochs):
                # Training with gradient descent
                for X, y, seq_length in tfe.Iterator(training_data):
                    grads = self.grads_fn(X, y, seq_length, True)
                    optimizer.apply_gradients(zip(grads, self.variables))

                # Check accuracy train dataset
                for X, y, seq_length in tfe.Iterator(training_data):
                    logits = self.predict(X, seq_length, False)
                    preds = tf.argmax(logits, axis=1)
                    train_acc(preds, y)
                self.history['train_acc'].append(train_acc.result().numpy())
                # Reset metrics
                train_acc.init_variables()

                # Check accuracy eval dataset
                for X, y, seq_length in tfe.Iterator(eval_data):
                    logits = self.predict(X, seq_length, False)
                    preds = tf.argmax(logits, axis=1)
                    eval_acc(preds, y)
                self.history['eval_acc'].append(eval_acc.result().numpy())
                # Reset metrics
                eval_acc.init_variables()

                # Print train and eval accuracy
                if (i == 0) | ((i + 1) % verbose == 0):
                    print('Train accuracy at epoch %d: ' % (i + 1), self.history['train_acc'][-1])
                    print('Eval accuracy at epoch %d: ' % (i + 1), self.history['eval_acc'][-1])

                # Check for early stopping
                if self.history['eval_acc'][-1] > best_acc:
                    best_acc = self.history['eval_acc'][-1]
                    count = early_stopping_rounds
                else:
                    count -= 1
                if count == 0:
                    break

def process_new_review(review):
    '''Function to process a new review.
       Args:
           review: original text review, string.
       Returns:
           indexed_review: sequence of integers, words correspondence
                           from word2idx.
           seq_length: the length of the review.
    '''
    indexed_review = re.sub(r'<[^>]+>', ' ', review)
    indexed_review = word_tokenize(indexed_review)
    indexed_review = [word2idx[word] if word in list(word2idx.keys()) else
                      word2idx['Unknown_token'] for word in indexed_review]
    indexed_review = indexed_review + [word2idx['End_token']]
    seq_length = len(indexed_review)
    return indexed_review, seq_length

def test():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    sent_dict = {0: 'negative', 1: 'positive'}
    review_score_10 = "I think Bad Apples is a great time and I recommend! I enjoyed the opening, which gave way for the rest of the movie to occur. The main couple was very likable and I believed all of their interactions. They had great onscreen chemistry and made me laugh quite a few times! Keeping the girls in the masks but seeing them in action was something I loved. It kept a mystery to them throughout. I think the dialogue was great. The kills were fun. And the special surprise gore effect at the end was AWESOME!! I won't spoil that part ;) I also enjoyed how the movie wrapped up. It gave a very urban legends type feel of \"did you ever hear the story...\". Plus is leaves the door open for another film which I wouldn't mind at all. Long story short, I think if you take the film for what it is; a fun little horror flick, then you won't be disappointed! HaPpY eArLy HaLLoWeEn!"
    review_score_4 = "A young couple comes to a small town, where the husband get a job working in a hospital. The wife which you instantly hate or dislike works home, at the same time a horrible murders takes place in this small town by two masked killers. Bad Apples is just your tipical B-horror movie with average acting (I give them that. Altough you may get the idea that some of the actors are crazy-convervative Christians), but the script is just bad, and that's what destroys the film."
    review_score_1 = "When you first start watching this movie, you can tell its going to be a painful ride. the audio is poor...the attacks by the \"girls\" are like going back in time, to watching the old rocky films, were blows never touched. the editing is poor with it aswell, example the actress in is the bath when her husband comes home, clearly you see her wearing a flesh coloured bra in the bath. no hints or spoilers, just wait till you find it in a bargain basket of cheap dvds in a couple of weeks"
    new_reviews = [review_score_10, review_score_4, review_score_1]
    scores = [10, 4, 1]
    with tf.device(device):
        for original_review, score in zip(new_reviews, scores):
            indexed_review, seq_length = process_new_review(original_review)
            indexed_review = tf.reshape(tf.constant(indexed_review), (1, -1))
            seq_length = tf.reshape(tf.constant(seq_length), (1,))
            logits = lstm_model.predict(indexed_review, seq_length, False)
            pred = tf.argmax(logits, axis=1).numpy()[0]
            print('The sentiment for the review with score %d was found to be %s'
                  % (score, sent_dict[pred]))




if __name__ == '__main__':
    train_dataset = tf.data.TFRecordDataset('dataset/aclImdb/train.tfrecords')
    train_dataset = train_dataset.map(parse_imdb_sequence).shuffle(buffer_size=10000)
    train_dataset = train_dataset.padded_batch(512, padded_shapes=([None], [], []))

    test_dataset = tf.data.TFRecordDataset('dataset/aclImdb/test.tfrecords')
    test_dataset = test_dataset.map(parse_imdb_sequence).shuffle(buffer_size=10000)
    test_dataset = test_dataset.padded_batch(512, padded_shapes=([None], [], []))

    # Read the word vocabulary
    word2idx = pickle.load(open('dataset/aclImdb/word2idx.pkl', 'rb'))

    checkpoint_directory = 'models_checkpoints/ImdbRNN/'
    device = 'gpu:0' if tfe.num_gpus() > 0 else 'cpu:0'
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    lstm_model = RNNModel(vocabulary_size=len(word2idx), device=device,
                          checkpoint_directory=checkpoint_directory)
    # lstm_model.fit(train_dataset, test_dataset, optimizer, num_epochs=10,
    #                early_stopping_rounds=5, verbose=1, train_from_scratch=True)
    # Save model
    # lstm_model.save_model()
    lstm_model.restore_model()
    test()