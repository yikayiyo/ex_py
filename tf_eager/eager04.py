# -*- encoding='utf-8' -*-
from nltk.tokenize import word_tokenize
import tensorflow as tf
import pandas as pd
import pickle
import random
import glob
# import nltk
import re

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

def imdb2tfrecords(path_data='dataset/aclImdb/train/', min_word_frequency=5,
                   max_words_review=700):
    '''
    处理数据保存为tfrecords
    Args:
        path_data: 数据路径
        min_word_frequency: 词典中保留词的最低词频
        max_words_review: 评论允许保留的最多单词数
    '''
    # 返回匹配的文件路径列表
    pos_files = glob.glob(path_data + 'pos/*')
    neg_files = glob.glob(path_data + 'neg/*')
    # 拼接两个列表
    filenames = pos_files + neg_files
    # 评论列表
    reviews = [open(filenames[i], 'r',encoding='utf-8').read() for i in range(len(filenames))]
    # 移除HTML标签
    reviews = [re.sub(r'<[^>]+>', ' ', review) for review in reviews]
    # 将每一条评论分词
    reviews = [word_tokenize(review) for review in reviews]
    # 计算每一条评论的长度
    len_reviews = [len(review) for review in reviews]
    # 展开为单词列表
    reviews = [word for review in reviews for word in review]
    # 统计词频
    word_frequency = pd.value_counts(reviews)
    # 词频大于minimal的词列表
    vocabulary = word_frequency[word_frequency >= min_word_frequency].index.tolist()
    # 添加 Unknown and End token.
    extra_tokens = ['Unknown_token', 'End_token']
    vocabulary += extra_tokens
    # 创建word2idx字典
    word2idx = {vocabulary[i]: i for i in range(len(vocabulary))}
    # 保存到Disk
    with open(path_data + 'word2idx.txt', 'wb') as f:
        pickle.dump(word2idx,f)

    def text2tfrecords(filenames, writer, vocabulary, word2idx,
                       max_words_review):
        '''
        解析评论，以tfrecords的形式写入磁盘
        Args:
            filenames: 评论路径
            writer: tfrecords的writer对象
            vocabulary:
            word2idx:
        '''
        # 打乱文件名
        random.shuffle(filenames)
        for filename in filenames:
            review = open(filename, 'r',encoding='utf-8').read()
            review = re.sub(r'<[^>]+>', ' ', review)
            review = word_tokenize(review)
            # 限制评论长度
            review = review[-max_words_review:]
            # 将单词用index代替
            review = [word2idx[word] if word in vocabulary else
                      word2idx['Unknown_token'] for word in review]
            indexed_review = review + [word2idx['End_token']]
            sequence_length = len(indexed_review)
            target = 1 if filename.split('/')[-2] == 'pos' else 0
            # 创建一个Sequence Example来存储数据
            ex = tf.train.SequenceExample()
            # 非字符串特征：长度，标签
            ex.context.feature['sequence_length'].int64_list.value.append(sequence_length)
            ex.context.feature['target'].int64_list.value.append(target)
            # 字符串特征
            token_indexes = ex.feature_lists.feature_list['token_indexes']
            for token_index in indexed_review:
                token_indexes.feature.add().int64_list.value.append(token_index)
            writer.write(ex.SerializeToString())

    ##########################################################################
    # Write data to tfrecords.This might take a while.
    ##########################################################################
    writer = tf.python_io.TFRecordWriter(path_data + '_dummy.tfrecords')
    text2tfrecords(filenames, writer, vocabulary, word2idx,
                   max_words_review)


def parse_imdb_sequence(record):
    '''
    Script to parse imdb tfrecords.

    Returns:
        token_indexes: sequence of token indexes present in the review.
        target: the target of the movie review.
        sequence_length: the length of the sequence.
    '''
    context_features = {
        'sequence_length': tf.FixedLenFeature([], dtype=tf.int64),
        'target': tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        'token_indexes': tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(record,
                                                                       context_features=context_features,
                                                                       sequence_features=sequence_features)

    return (sequence_parsed['token_indexes'], context_parsed['target'],
            context_parsed['sequence_length'])
if __name__ == '__main__':
    # imdb2tfrecords()
    record = 'dataset/aclImdb/train/dummy.tfrecords'
    print(parse_imdb_sequence(record=record))