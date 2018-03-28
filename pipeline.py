"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2018-2019
"""

import tensorflow as tf
from tensorflow.python.data import Iterator
from tensorflow.python.data import TFRecordDataset


class Vector_Pipeline(object):
    def __init__(self, train_tfrecord, val_tfrecord, batch_size,
                 buffer_size=50):
        """
        Load vectors of integers from tfrecord files.

        note: pads batches w/ 0 so all sequences match the longest example
              adds 1 to all other values so that 0 is unique to padding

        :param train_tfrecord: path to tfrecord file
        :param val_tfrecord: path to tfrecord file
        :param batch_size: batch size
        :param buffer_size: examples in buffer for shuffling
        """
        assert isinstance(train_tfrecord, str)
        assert isinstance(val_tfrecord, str)
        assert isinstance(batch_size, int)
        assert batch_size>0
        assert buffer_size>0

        if buffer_size<batch_size:
            buffer_size = batch_size

        def dataset(tfrecord):
            ds = TFRecordDataset(tfrecord)
            def parse(x):
                example = tf.parse_single_example(
                    x, features={'data': tf.VarLenFeature(tf.int64)})
                example = tf.cast(example['data'].values, tf.int32)+1
                return example
            ds = ds.map(parse, num_parallel_calls=8)
            return ds.padded_batch(batch_size,
                                   padded_shapes=(tf.TensorShape([None])),
                                   padding_values=0)

        train_dataset = dataset(train_tfrecord).shuffle(buffer_size)
        val_dataset = dataset(val_tfrecord)
        iterator = Iterator.from_structure(
            train_dataset.output_types,train_dataset.output_shapes)
        self.init_train = iterator.make_initializer(train_dataset,
                                                    name='init_train')
        self.init_val = iterator.make_initializer(val_dataset,
                                                  name='init_val')
        self.output = iterator.get_next()