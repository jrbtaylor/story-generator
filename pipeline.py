"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import tensorflow as tf
from tensorflow.python.data import Iterator
from tensorflow.python.data import TFRecordDataset


class Vector_Pipeline(object):
    def __init__(self, train_tfrecord, val_tfrecord, batch_size):
        """
        Load vectors of integers from tfrecord files.

        note: pads batches w/ -1 so all sequences match the longest example

        :param train_tfrecord: path to tfrecord file
        :param val_tfrecord: path to tfrecord file
        :param batch_size: batch size
        """
        assert isinstance(train_tfrecord, str)
        assert isinstance(val_tfrecord, str)
        assert isinstance(batch_size, int)
        assert batch_size>0

        def dataset(tfrecord):
            ds = TFRecordDataset(tfrecord)
            ds = ds.shuffle(10*batch_size)
            def parse(x):
                example = tf.parse_single_example(x,
                    features={'data': tf.VarLenFeature([], tf.int64)})
                return example['data']
            ds = ds.map(parse, num_parallel_calls=8)
            return ds.padded_batch(batch_size, padded_shapes=[None],
                                   padding_values=-1)

        train_dataset = dataset(train_tfrecord)
        val_dataset = dataset(val_tfrecord)
        iterator = Iterator.from_structure(
            train_dataset.output_types,train_dataset.output_shapes)
        self.init_train = iterator.make_initializer(train_dataset,
                                                    name='init_train')
        self.init_val = iterator.make_initializer(val_dataset,
                                                  name='init_val')
        self.output = iterator.get_next()[0]