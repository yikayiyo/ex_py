'''read_data_in_batches_from_tfrecords'''

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

def part1():
    '''变长序列'''
    #读取一批数据，将序列填充为批数据中最长序列的长度

    batch_size = 2
    dataset = tf.data.TFRecordDataset('dataset/dummy_text/dummy.tfrecords')
    dataset = dataset.map(parse_imdb_sequence).shuffle(buffer_size=10000)
    # 只有第一个字段需要填充，长度未知故为[None]
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [], []))
    for review, target, sequence_length in tfe.Iterator(dataset):
        print(review,target,sequence_length)
        print('----------------------------')


def parse_imdb_sequence(record):
    '''
    处理影评tfrecords.
    Returns:
        token_indexes: 文本中单词用index表示.
        target: 影评的标签.
        sequence_length: 影评长度.
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



def parser(record):
    '''
    Function to parse a TFRecords example.

    Returns:
        img:
        label: 标签.
    '''
    features = {'image': tf.FixedLenFeature((), tf.string),
                'label': tf.FixedLenFeature((), tf.int64)}
    parsed = tf.parse_single_example(record, features)
    img = tf.image.decode_image(parsed['image'])
    return img, parsed['label']


def part2():
    batch_size = 5
    dataset = tf.data.TFRecordDataset('dataset/dummy_images/dummy.tfrecords')
    dataset = dataset.map(parser).shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    # 迭代处理批数据，这里只是简单地打印
    emotion_cat = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
    for image, label in tfe.Iterator(dataset):
        #1行5列
        f, axarr = plt.subplots(1, int(image.shape[0]), figsize=(14, 6))
        for i in range(image.shape[0]):
            axarr[i].imshow(image[i,:,:,0], cmap='gray')
            axarr[i].set_title('Emotion: %s' %emotion_cat[label[i].numpy()])
        plt.show()
if __name__ == '__main__':
    part1()
    # part2()