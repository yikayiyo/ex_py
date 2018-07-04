import tensorflow as tf
import tensorflow.contrib.eager as tfe
import glob
tfe.enable_eager_execution()

# Get the meaning of each emotion index
emotion_cat = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}


def img2tfrecords(path_data='dataset/dummy_images/', image_format='jpeg'):
    ''' transfer raw images, along with their target labels, to TFRecords.
        Args:
            path_data: the location of the raw images
            image_format: the format of the raw images (e.g. 'png', 'jpeg')
    '''

    def _int64_feature(value):
        '''Helper function.'''
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        '''Helper function.'''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # 获取路径下的所有文件名
    filenames = glob.glob(path_data + '*' + image_format)

    # 创建一个TFRecordWriter
    writer = tf.python_io.TFRecordWriter(path_data + 'dummy.tfrecords')

    # 迭代文件写入TFrecords
    for filename in filenames:
        # 读入原始img
        img = tf.read_file(filename).numpy()
        # 从文件名获取标签
        label = int(filename.split('_')[-1].split('.')[0])
        # (image, label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'image': _bytes_feature(img)}))
        # 写入序列化样例
        writer.write(example.SerializeToString())
    writer.close()


def parser(record):
    '''Function to parse a TFRecords example'''

    # 定义特征
    features = {'image': tf.FixedLenFeature((), tf.string),
                'label': tf.FixedLenFeature((), tf.int64)}

    # 解析样例
    parsed = tf.parse_single_example(record, features)

    # 解码获取img
    img = tf.image.decode_image(parsed['image'])

    return img, parsed['label']

if __name__ == '__main__':
    # Transfer raw data to TFRecords
    # img2tfrecords()

    file_name = 'dataset/dummy_images/dummy.tfrecords'
    # error :Readers are not supported when eager execution is enabled
    # _,record = tf.TFRecordReader().read(file_name)
    dataset = tf.data.TFRecordDataset(file_name)
    dataset.map(parser)
    print(dataset)
