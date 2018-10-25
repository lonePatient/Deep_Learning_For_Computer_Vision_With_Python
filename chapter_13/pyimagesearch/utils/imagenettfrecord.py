#encoding:utf-8
from os import path
import tensorflow as tf

class ImageNetTfrecord(object):
    def __init__(self,tfrecord_name):
        self.tfrecord_name = tfrecord_name
        self.tfwriter = tf.python_io.TFRecordWriter(self.tfrecord_name)

    def _int64_feature(self,value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(self,value):
        """Wrapper for inserting float features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_feature(self,value):
        """Wrapper for inserting bytes features into Example proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _process_image(self,filename):
        """Process a single image file."""
        with tf.gfile.FastGFile(filename, 'rb') as f:
            image_data = f.read()
            return image_data

    def _save_one(self,label,filename,isTrain=True):
        image_data = self._process_image(filename)
        name = path.split(filename)[-1]
        if isTrain:
            example = tf.train.Example(features=tf.train.Features(feature={
                        'image': self._bytes_feature(tf.compat.as_bytes(image_data)),
                        'label': self._int64_feature(label),
                        'name': self._bytes_feature(tf.compat.as_bytes(name))
                    }))
            self.tfwriter.write(example.SerializeToString())

        else:
            label = int(-1)
            example = tf.train.Example(features=tf.train.Features(feature={
                    'image': self._bytes_feature(tf.compat.as_bytes(image_data)),
                    'label': self._int64_feature(label),
                    'name': self._bytes_feature(tf.compat.as_bytes(name))
                }))
            self.tfwriter.write(example.SerializeToString())

